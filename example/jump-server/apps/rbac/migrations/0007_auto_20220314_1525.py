# Generated by Django 3.1.14 on 2022-03-14 07:25

from django.db import migrations


def migrate_old_permissions(apps, *args):
    ContentType = apps.get_model('rbac', 'ContentType')
    content_type_delete_required = [
        ('common', 'permission'),
        ('applications', 'databaseapp'),
        ('applications', 'k8sapp'),
        ('applications', 'remoteapp'),
        ('perms', 'databaseapppermission'),
        ('perms', 'k8sapppermission'),
        ('perms', 'remoteapppermission'),
        ('authentication', 'loginconfirmsetting'),
    ]
    for app, model in content_type_delete_required:
        ContentType.objects.filter(app_label=app, model=model).delete()


class Migration(migrations.Migration):

    dependencies = [
        ('rbac', '0006_auto_20220310_0616'),
    ]

    operations = [
        migrations.RunPython(migrate_old_permissions)
    ]
