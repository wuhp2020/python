# Generated by Django 3.1.13 on 2021-12-01 11:01

from django.db import migrations

from rbac.builtin import BuiltinRole


def migrate_system_role_binding(apps, schema_editor):
    db_alias = schema_editor.connection.alias
    user_model = apps.get_model('users', 'User')
    role_binding_model = apps.get_model('rbac', 'SystemRoleBinding')
    users = user_model.objects.using(db_alias).all()

    role_bindings = []
    for user in users:
        role = BuiltinRole.get_system_role_by_old_name(user.role)
        role_binding = role_binding_model(scope='system', user_id=user.id, role_id=role.id)
        role_bindings.append(role_binding)
    role_binding_model.objects.bulk_create(role_bindings, ignore_conflicts=True)


def migrate_org_role_binding(apps, schema_editor):
    db_alias = schema_editor.connection.alias
    org_member_model = apps.get_model('orgs', 'OrganizationMember')
    role_binding_model = apps.get_model('rbac', 'RoleBinding')
    members = org_member_model.objects.using(db_alias).all()

    role_bindings = []
    for member in members:
        role = BuiltinRole.get_org_role_by_old_name(member.role)
        role_binding = role_binding_model(
            scope='org',
            user_id=member.user.id,
            role_id=role.id,
            org_id=member.org.id
        )
        role_bindings.append(role_binding)
    role_binding_model.objects.bulk_create(role_bindings)


class Migration(migrations.Migration):

    dependencies = [
        ('rbac', '0003_auto_20211130_1037'),
    ]

    operations = [
        migrations.RunPython(migrate_system_role_binding),
        migrations.RunPython(migrate_org_role_binding)
    ]
